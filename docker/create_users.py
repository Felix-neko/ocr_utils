# Идемпотентно создаёт пользователей, организацию и членства.
# Запускается ВНУТРИ контейнера cvat_server через:
#   docker exec -i cvat_server python3 /home/django/manage.py shell < create_users.py
# Параметры берутся из переменных окружения (передаём их в docker exec).

import os

from django.contrib.auth import get_user_model
from django.utils import timezone

from cvat.apps.organizations.models import Membership, Organization

User = get_user_model()


def ensure_user(username, password, email, is_superuser=False):
    user, created = User.objects.get_or_create(
        username=username,
        defaults={"email": email},
    )
    user.email = email
    user.is_active = True
    user.is_staff = is_superuser
    user.is_superuser = is_superuser
    user.set_password(password)  # каждый запуск синхронизирует пароль с .env
    user.save()

    # Отмечаем e-mail подтверждённым (на случай ACCOUNT_EMAIL_VERIFICATION!="none").
    try:
        from allauth.account.models import EmailAddress

        EmailAddress.objects.update_or_create(
            user=user,
            email=email,
            defaults={"verified": True, "primary": True},
        )
    except Exception as exc:  # noqa: BLE001
        print(f"  (пропущено создание EmailAddress: {exc})")

    print(f"  пользователь {username!r}: {'создан' if created else 'обновлён'} "
          f"(superuser={is_superuser})")
    return user


def ensure_membership(user, organization, role):
    membership, created = Membership.objects.get_or_create(
        user=user,
        organization=organization,
        defaults={"role": role, "is_active": True, "joined_date": timezone.now()},
    )
    changed = False
    if membership.role != role:
        membership.role = role
        changed = True
    if not membership.is_active:
        membership.is_active = True
        changed = True
    if membership.joined_date is None:
        membership.joined_date = timezone.now()
        changed = True
    if changed:
        # Запись роли CVAT сопровождает событием в аналитику (vector), а она у нас выключена
        # профилем. Отправка падает УЖЕ ПОСЛЕ коммита: роль записана, а скрипт валится с
        # ConnectionError, и вместе с ним — весь up.sh, потому что там set -e. Ловим ровно
        # это и ровно здесь: сама запись состоялась, а недоставленное событие аналитики нам
        # безразлично.
        from requests.exceptions import ConnectionError as RequestsConnectionError

        try:
            membership.save()
        except RequestsConnectionError as exc:
            print(f"  (аналитика недоступна, событие не отправлено: {exc.__class__.__name__}) ")
    print(f"  членство {user.username!r} в {organization.slug!r}: роль={role} "
          f"({'создано' if created else 'обновлено'})")


admin_user = os.environ["ADMIN_USER"]
admin_pass = os.environ["ADMIN_PASS"]
ann_user = os.environ["ANN_USER"]
ann_pass = os.environ["ANN_PASS"]
# Роль разметчика в организации. Задаётся из .env, а не зашита в код: worker видит только
# НАЗНАЧЕННЫЕ ему джобы, и вкладки Projects и Tasks у него пусты всегда. Кому нужен обзор
# по годам — тому нужен maintainer, и это решение обязано переживать перезапуск: скрипт
# приводит роль к заданной при каждом up.sh, и зашитый worker молча откатывал бы её назад.
ann_role = os.environ.get("ANN_ROLE", "worker").strip().lower()
allowed_roles = {"worker": Membership.WORKER, "supervisor": Membership.SUPERVISOR, "maintainer": Membership.MAINTAINER}
if ann_role not in allowed_roles:
    raise SystemExit(f"ANN_ROLE={ann_role!r}: допустимы {', '.join(sorted(allowed_roles))}")
org_slug = os.environ["ORG_SLUG"]
org_name = os.environ["ORG_NAME"]

print("Пользователи:")
admin = ensure_user(admin_user, admin_pass, "admin@example.com", is_superuser=True)
annotator = ensure_user(ann_user, ann_pass, "user@example.com", is_superuser=False)

org, created = Organization.objects.get_or_create(
    slug=org_slug,
    defaults={"name": org_name, "owner": admin},
)
if org.name != org_name or org.owner_id != admin.id:
    org.name = org_name
    org.owner = admin
    org.save()
print(f"Организация {org_slug!r} ({org_name!r}): {'создана' if created else 'обновлена'}")

print("Членства:")
ensure_membership(admin, org, Membership.OWNER)
ensure_membership(annotator, org, allowed_roles[ann_role])

print("create_users.py: готово.")
